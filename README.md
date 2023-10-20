## Fuzzy Text Search Plugin

This plugin is a Python plugin that allows you to semantically search through your text blocks (from Optical Character Recognition) in your dataset.

It uses a Qdrant index, with the GTE-base model from Hugging Face's Sentence Transformers library.

## Usage

You will need to have text blocks in your dataset. You can do this with the [PyTesseract OCR plugin](https://github.com/jacobmarks/pytesseract-ocr-plugin).

Create a vector index for your text blocks with the `create_text_index` operator. You can then use the `fuzzy_search_text` operator to search through your text blocks.

If you have multiple detections with text blocks, you can create multiple indexes. The index is stored in Qdrant with the collection name `<dataset_name>_fuzzy_<field_name>`. When you use the `fuzzy_search_text` operator, you can specify which index to use.

## Installation

Download the plugin with the following command:

```shell
fiftyone plugins download https://github.com/jacobmarks/fuzzy-search-plugin
```

You will need to install the Sentence Transformers library, and the Qdrant client Python library, which can be achieved with

```shell
fiftyone plugins requirements @jacobmarks/fuzzy_search --install
```

You will also need to have a Qdrant instance running. You can do this with Docker once you have your Docker daemon running:

```shell
docker run -p "6333:6333" -p "6334:6334" -d qdrant/qdrant
```

## Using with PyTesseract OCR Plugin

This "fuzzy search" plugin is in many ways analogous to the [keyword search plugin](https://github.com/jacobmarks/keyword-search-plugin), and is likewise designed to be used with the [PyTesseract OCR plugin](https://github.com/jacobmarks/pytesseract-ocr-plugin).

You can install the PyTesseract OCR plugin with the following command:

```shell
fiftyone plugins download https://github.com/jacobmarks/pytesseract-ocr-plugin
```

## Operators

### `create_text_index`

![thesis_create_index](https://github.com/jacobmarks/fuzzy-search-plugin/assets/12500356/1660d9e8-c7b8-4e58-843f-f016555c451e)

**Description**: Create a Qdrant index for the specified text field within a detections label field.

### `fuzzy_search_text`

![thesis_search](https://github.com/jacobmarks/fuzzy-search-plugin/assets/12500356/63082f25-640c-45ef-8e77-38e27fba0269)

**Description**: Semantically, of "fuzzilly" search for text in your dataset. Only labels matching your query will be shown.

You can specify the number of results to return, and the threshold for the similarity score.
