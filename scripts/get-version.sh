#!/usr/bin/env bash
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)

LATEST_VERSION_TAG=$(git describe --tags --match="v[0-9]*" external --abbrev=0)

LAST_MAJOR_MINOR=$(echo $LATEST_VERSION_TAG | sed -r 's/v([0-9]+)\.([0-9]+)\.[0-9]+/\1.\2/g')
NEXT_MAJOR_MINOR=$(scripts/next-version.sh $LAST_MAJOR_MINOR)

if [[ $BRANCH_NAME == "main" ]]; then
    VERSION=main-$NEXT_MAJOR_MINOR-$(git rev-parse --short HEAD)
    elif [[ $BRANCH_NAME == "external" ]]; then
    # valid tag in this case is the version tag
    # ... but this should be a manual tagging step (for now at least)
    echo "Versioning must be manual on external branch"
    exit 1
    elif [[ $BRANCH_NAME =~ release/v[0-9]* ]]; then
    # ignore NEXT_MAJOR_MINOR and use the branch name version
    RELEASE_BRANCH_VERSION=$(echo $BRANCH_NAME | sed -r 's/release\///g')
    VERSION=rc-$RELEASE_BRANCH_VERSION-$(git rev-parse --short HEAD)
else
    # replace "_" with "-"
    BRANCH_NAME=$(echo "$BRANCH_NAME" | sed -r 's/[_:]/-/g')
    # only allow a single tag for each feature branch to save docker image space
    VERSION=f-$BRANCH_NAME-$NEXT_MAJOR_MINOR
fi

echo $VERSION
