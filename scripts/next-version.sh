#!/usr/bin/env bash

# script returns next semantic version.
version=$1
# Input version must be either major.minor.patch or major.minor.
# Script will return the next patch or minor version respectively
# so:
# next-version.sh 1.2.3 will return 1.2.4
# next-version.sh 1.2 will return 1.3

if [[ $version =~ ^v?[0-9]+\.[0-9]+ ]]; then
	# edited from https://unix.stackexchange.com/questions/23174/increment-number-in-bash-variable-string:
	[[ $version =~ (.*[^0-9])([0-9]+)$ ]] && version="${BASH_REMATCH[1]}$((${BASH_REMATCH[2]} + 1))"
	echo "v$version"
else
	echo "Invalid version tag: '$version'"
	exit 1

fi
