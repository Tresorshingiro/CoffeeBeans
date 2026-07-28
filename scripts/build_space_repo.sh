#!/usr/bin/env bash
# Assemble a Hugging Face Space repo from this repository.
#
# A Space is not just this repo with a different remote. HF builds ./Dockerfile
# at the Space root and reads its YAML frontmatter from ./README.md — this repo
# keeps the Space Dockerfile under docker/ and its frontmatter in README_HF.md,
# because the root Dockerfile and README belong to docker-compose and to the
# graders. So the Space gets its own generated directory.
#
# It also ships only what the container actually needs: the notebook, tests,
# locust results, SavedModel export and .h5 checkpoint all matter to the GitHub
# repo and to grading, but they are dead weight in a Space image and would drag
# another 41MB through Git LFS for nothing.
#
#     ./scripts/build_space_repo.sh [dest]     # default: ../CoffeeBeans-space
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$(cd "$SRC/.." && pwd)/CoffeeBeans-space}"

echo "source: $SRC"
echo "dest:   $DEST"

mkdir -p "$DEST"
# Keep .git so an existing Space clone keeps its history and remote, and keep
# .gitattributes so HF's LFS defaults survive a regenerate.
find "$DEST" -mindepth 1 -maxdepth 1 ! -name .git ! -name .gitattributes \
    -exec rm -rf {} +

mkdir -p "$DEST/docker" "$DEST/models" "$DEST/data"

for item in src api ui scripts requirements.txt; do
    cp -r "$SRC/$item" "$DEST/$item"
done
rm -f "$DEST/scripts/build_space_repo.sh"      # no reason to ship the generator

cp "$SRC/docker/nginx.space.conf" "$DEST/docker/"
cp "$SRC/docker/start.sh"         "$DEST/docker/"

# HF looks for these two at the root, under these exact names.
cp "$SRC/docker/Dockerfile.space" "$DEST/Dockerfile"
cp "$SRC/README_HF.md"            "$DEST/README.md"

cp "$SRC/models/coffee_model.keras" "$DEST/models/"
cp -r "$SRC/data/train" "$SRC/data/test" "$DEST/data/"
cp "$SRC/data/insights.json" "$DEST/data/"

# HF rejects files over 10MB that are not LFS-tracked. A cloned Space ships a
# 35-line .gitattributes covering *.h5, *.pb and friends — keep it and append
# only what is missing, rather than replacing a well-tested default with one
# line. coffee_model.keras is the only >10MB file left in a lean Space repo.
touch "$DEST/.gitattributes"
if ! grep -q '^\*\.keras ' "$DEST/.gitattributes" 2>/dev/null; then
    echo '*.keras filter=lfs diff=lfs merge=lfs -text' >> "$DEST/.gitattributes"
fi

cat > "$DEST/.dockerignore" <<'EOF'
.git/
**/__pycache__/
data/pending/
data/app.db*
EOF

cat > "$DEST/.gitignore" <<'EOF'
__pycache__/
*.py[cod]
data/pending/
data/app.db*
models/coffee_model_*.keras
EOF

echo
echo "assembled $(find "$DEST" -type f -not -path '*/.git/*' | wc -l) files, $(du -sh --exclude=.git "$DEST" | cut -f1)"
echo "over 10MB (must be LFS-tracked):"
find "$DEST" -type f -size +10M -not -path '*/.git/*' -printf '  %s  %p\n' | numfmt --field=1 --to=iec
