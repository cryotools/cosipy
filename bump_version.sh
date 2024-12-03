#!/bin/bash
# 2024 Nicolas Gampierakis.

# Applies semantic versioning across project

DIRECTORY=""
VERBOSE=0
BRANCH=""

DisplayHelp() {
    intro_tag="$(basename "$0") [-h] -- semantic versioning tool"
    options="OPTIONS
    -h, --help              Display help.
    -d, --directory [path]  Path to target directory, relative to current
                                working directory.
    -v, --verbose           Verbosity flag.
    "
    printf "\n%s\n\n%s\n" "$intro_tag" "$options"
}

#######################################
# Get project version number
#
# Arguments:
#   Keyword for version number in file.
#   File path.
#######################################
get_version_number() {
    match_string="${1}"
    file_name="${2}"
    version_number=$(grep -oP '\b'"${match_string}"'\s*=\s*"\K.*?(?=")' "${file_name}")
}

#######################################
# Set project version number
#
# Arguments:
#   Project root directory.
#######################################
set_version_number() {
    toml_file="${1}pyproject.toml"
    docs_file="${1}docs/source/conf.py"
    version_number=""
    get_version_number "version" "${toml_file}"
    current_py_version="${version_number}"
    get_version_number "release" "${docs_file}"
    current_docs_version="${version_number}"
    current_version="${current_py_version}"

    new_version=${BRANCH#"release-"}
    if [ "${new_version}" != "${current_version}" ]; then
        printf "%s\n" "Current branch: ${BRANCH}" >&2
        printf "%s\n" "Previous version number: ${current_version}" >&2
        printf "%s\n" "New version number: ${new_version}" >&2
        sed -i "s/${current_py_version}/${new_version}/g" "${toml_file}"
        sed -i "s/${current_docs_version}/${new_version}/g" "${docs_file}"
        git add "${toml_file}" "${docs_file}"
        git commit -m "build: bump version number to ${new_version}"
    else
        printf "%s\n" "Current and new version numbers are identical: ${current_version}" >&2
    fi
}

ARGS=$(getopt -o "hd:v" --long "help,directory:,verbose" -- "$@") || exit
eval "set -- $ARGS"
while true; do
    case $1 in
    -h | --help)
        DisplayHelp
        exit 0
        ;;
    -d | --directory)
        DIRECTORY=$2
        shift 2
        ;;
    -v | --verbose)
        ((VERBOSE++))
        shift
        ;;
    --)
        shift
        break
        ;;
    *) exit 1 ;; # error
    esac
done
remaining=("$@")

if [[ ! $DIRECTORY ]]; then
    DIRECTORY="${PWD}/"
else
    DIRECTORY="${PWD}/${DIRECTORY}"
fi
readonly DIRECTORY

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != *"release-"* ]]; then
    echo "Please switch to a release branch."
    exit 1
fi

set_version_number "${DIRECTORY}"
