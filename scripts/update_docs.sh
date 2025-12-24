#!/usr/bin/env bash

set -e
DOCS_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../docs" && pwd)

# Update the rendered diagrams in the docs.
dot -Tpng $DOCS_DIR/dot/equivfusion.dot > $DOCS_DIR/includes/img/equivfusion.png
dot -Tsvg $DOCS_DIR/dot/equivfusion.dot > $DOCS_DIR/includes/img/equivfusion.svg

