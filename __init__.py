"""Fuzzy Search plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import json
import os

from bson import json_util

import fiftyone as fo
import fiftyone.core.utils as fou
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone import ViewField as F

from sentence_transformers import SentenceTransformer

import qdrant_client as qc
import qdrant_client.http.models as qmodels


def _to_qdrant_id(_id):
    return _id + "00000000"


def _to_qdrant_ids(ids):
    return [_to_qdrant_id(_id) for _id in ids]


def _to_fiftyone_id(qid):
    return qid.replace("-", "")[:-8]


def _get_model():
    return SentenceTransformer("thenlper/gte-base")


def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )


def serialize_view(view):
    return json.loads(json_util.dumps(view._serialize()))


def _create_index(ctx):
    dataset = ctx.dataset
    detection_field = ctx.params.get("detection_field", None)
    text_field = ctx.params.get("text_field", None)
    collection_name = f"{dataset.name.lower().replace(' ', '_').replace('-', '_')}_fuzzy_{detection_field}"

    embeddings, sample_ids, label_ids = [], [], []
    model = _get_model()

    view = dataset.exists(detection_field)

    for sample in view.iter_samples(progress=True):
        dets = sample[detection_field].detections
        for det in dets:
            sample_ids.append(sample.id)
            label_ids.append(det.id)
            embeddings.append(model.encode(det[text_field]))

    embeddings = [e.tolist() for e in embeddings]

    batch_size = 100

    client = qc.QdrantClient()
    vectors_config = qmodels.VectorParams(
        size=768,
        distance=qmodels.Distance.COSINE,
    )

    client.recreate_collection(
        collection_name=collection_name, vectors_config=vectors_config
    )

    for _embeddings, _ids, _sample_ids in zip(
        fou.iter_batches(embeddings, batch_size),
        fou.iter_batches(label_ids, batch_size),
        fou.iter_batches(sample_ids, batch_size),
    ):
        client.upsert(
            collection_name=collection_name,
            points=qmodels.Batch(
                ids=_to_qdrant_ids(_ids),
                payloads=[{"sample_id": _id} for _id in _sample_ids],
                vectors=_embeddings,
            ),
        )


def _get_detections_fields(dataset):
    fields = []
    for field in dataset.get_field_schema().keys():
        view = dataset.exists(field)
        if len(view) == 0:
            continue
        if "Detections" in str(type(view.first()[field])):
            fields.append(field)
    return fields


def _get_text_field_name(dataset, detection_field):
    if dataset.distinct(f"{detection_field}.detections.label") != ["text"]:
        return "label"

    view = dataset.exists(detection_field)
    sample = view.first()
    det = sample[f"{detection_field}.detections"][0]

    subfield_names = det.field_names
    for sf in subfield_names:
        if "'str'" in str(type(det[sf])) and sf not in ["label", "id"]:
            return sf
    else:
        return None


def _handle_index_field(ctx, inputs):
    dataset = ctx.dataset
    fields = _get_detections_fields(dataset)
    if len(fields) == 0:
        inputs.view(
            "warning",
            types.Warning(
                label="No candidate fields",
                description=(
                    "Cannot find any fields with detections. "
                    "You can run OCR on your dataset using the PyTesseract "
                    "OCR plugin: https://github.com/jacobmarks/pytesseract-ocr-plugin"
                ),
            ),
        )
    else:
        field_choices = types.RadioGroup()

        for field in fields:
            field_choices.add_choice(field, label=field)

        if "pt_block_predictions" in fields:
            _default = "pt_block_predictions"
        else:
            _default = fields[0]

        inputs.enum(
            "detection_field",
            field_choices.values(),
            label="Detections field",
            description="Select the field containing the detections to index",
            view=types.DropdownView(),
            required=True,
            default=_default,
        )

        detection_field = ctx.params.get("detection_field", _default)
        text_field = _get_text_field_name(dataset, detection_field)
        ctx.params["text_field"] = text_field

        if text_field is None:
            inputs.view(
                "warning",
                types.Warning(
                    label="No text field",
                    description=(
                        "Cannot find any text fields in the selected field. "
                        "You can run OCR on your dataset using the PyTesseract "
                        "OCR plugin:"
                    ),
                ),
            )
        else:
            inputs.view(
                "text_field_message",
                types.Header(
                    label=f"Text field: {text_field}",
                    description="Executing this operation will create a vector index for this field",
                    divider=False,
                ),
            )


class CreateGTEIndex(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="create_text_index",
            label="Fuzzy Search: Create vector index for text blocks with GTE",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.view(
            "header",
            types.Header(
                label="Create vector index",
                description="Create Qdrant index for text blocks with GTE model",
                divider=True,
            ),
        )
        _handle_index_field(ctx, inputs)
        _execution_mode(ctx, inputs)
        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        _create_index(ctx)
        ctx.trigger("reload_dataset")


def _get_matching_collections(ctx):
    dsn = ctx.dataset.name.lower().replace(" ", "_").replace("-", "_")
    prefix = f"{dsn}_fuzzy_"

    client = qc.QdrantClient()
    collections = client.get_collections().collections
    return [c.name for c in collections if c.name.startswith(prefix)]


def _extract_detections_field(collection_name):
    return collection_name.split("_fuzzy_")[-1]


def _run_query(ctx):
    collection_name = ctx.params.get("collection_name")
    detection_field = _extract_detections_field(collection_name)

    current_ids = ctx.view.values(
        f"{detection_field}.detections.id", unwind=True
    )

    _filter = qmodels.Filter(
        must=[qmodels.HasIdCondition(has_id=_to_qdrant_ids(current_ids))]
    )

    query_text = ctx.params.get("query")
    threshold = ctx.params.get("threshold")
    k = ctx.params.get("k")

    model = _get_model()
    query = model.encode(query_text)

    client = qc.QdrantClient()
    results = client.search(
        collection_name=collection_name,
        query_vector=query,
        with_payload=False,
        limit=k,
        query_filter=_filter,
        score_threshold=threshold,
    )

    label_ids = [_to_fiftyone_id(sc.id) for sc in results]
    view = ctx.dataset.select_labels(ids=label_ids)
    return view


class FuzzySearch(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="fuzzy_search_text",
            label="Fuzzy Search: search text blocks semantically",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Semantically search text blocks",
                icon="/assets/icon.svg",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Fuzzy Search", description="Semantically search text blocks"
        )

        valid_collections = _get_matching_collections(ctx)
        if len(valid_collections) == 0:
            inputs.view(
                "warning",
                types.Warning(
                    label="No available index",
                    description=(
                        "No valid index found. You can create an index "
                        "using the `create_text_index` operator"
                    ),
                ),
            )
            return types.Property(inputs, view=form_view)
        elif len(valid_collections) == 1:
            collection_name = valid_collections[0]
            ctx.params["collection_name"] = collection_name
            detections_field = _extract_detections_field(collection_name)
            text_field = _get_text_field_name(ctx.dataset, detections_field)

            inputs.view(
                "index_text_field_message",
                types.Header(
                    label=f"Index for text field {detections_field}.detections.{text_field}",
                ),
            )
        else:
            detection_fields = [
                _extract_detections_field(c) for c in valid_collections
            ]
            text_fields = [
                _get_text_field_name(ctx.dataset, df)
                for df in detection_fields
            ]

            collection_choices = types.Dropdown(multiple=False)

            for cn, df, tf in zip(
                valid_collections, detection_fields, text_fields
            ):
                collection_choices.add_choice(cn, label=f"{df} ({tf})")

            inputs.enum(
                "collection_name",
                collection_choices.values(),
                label="Index",
                description="Select the index to search",
                view=collection_choices,
                required=True,
            )

        inputs.str("query", label="Query", required=True)

        inputs.int(
            "k",
            label="num results",
            default=20,
        )

        inputs.float(
            "threshold",
            label="Threshold score for matching",
            default=0.8,
        )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        view = _run_query(ctx)
        ctx.trigger(
            "set_view",
            params=dict(view=serialize_view(view)),
        )
        return


def register(plugin):
    plugin.register(CreateGTEIndex)
    plugin.register(FuzzySearch)
