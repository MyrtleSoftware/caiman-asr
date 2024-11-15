#!/usr/bin/env bash
set -Eeuo pipefail

mdbook-admonish install

# Make a copy of the code in a directory not visible
# to the host. Hence this copy can be modified
# without affecting the host's copy.
git config --global --add safe.directory /hidden_code
cp -r /code /hidden_code

cd /hidden_code/docs

current_commit=$(git rev-parse HEAD)

versions=$(git tag | grep -P '^v\d+\.\d+\.\d+$')

sorted_versions=$(echo $versions | tr ' ' '\n' | sort -V)

# Filter for the versions where mdbook docs existed
versions_with_docs=""
for tag in $sorted_versions; do
	git checkout $tag
	if [ -f src/SUMMARY.md ]; then
		versions_with_docs+=" $tag"
	fi
done

# Must build the latest docs first,
# else it overwrites the others
git checkout $current_commit
# Update the page pointing to older docs:
for tag in $versions_with_docs; do
	echo "- [$tag]($tag)" >>src/versions.md
done
mdbook build

# Restore that page, else we can't checkout other commits
git restore src/versions.md

# Build the older docs
for tag in $versions_with_docs; do
	git checkout $tag
	mdbook build --dest-dir=book/$tag
done

# cp will nest the directories if
# the destination directory exists,
# so delete it
if [ -d /code/docs/book ]; then
	rm -r /code/docs/book
fi

# Give the built book back to the host
cp -r /hidden_code/docs/book /code/docs/
