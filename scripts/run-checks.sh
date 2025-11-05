#!/usr/bin/env bash
# Pre-commit helper script
# Usage: ./scripts/run-checks.sh [OPTIONS]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Parse arguments
QUICK=false
VERBOSE=false
FIX=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--quick)
            QUICK=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--fix)
            FIX=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -q, --quick    Run only fast checks (skip mypy, bandit)"
            echo "  -v, --verbose  Show detailed output"
            echo "  -f, --fix      Auto-fix issues when possible"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if in git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository!"
    exit 1
fi

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    print_warning "pre-commit not found in PATH, using uv run..."
    PRECOMMIT_CMD="uv run pre-commit"
else
    PRECOMMIT_CMD="pre-commit"
fi

print_header "🔍 Running Code Quality Checks"

# Install pre-commit hooks if not already done
if [ ! -f .git/hooks/pre-commit ]; then
    print_warning "Pre-commit hooks not installed, installing now..."
    $PRECOMMIT_CMD install
    print_success "Pre-commit hooks installed"
fi

# Build command
CMD="$PRECOMMIT_CMD run --all-files"

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

if [ "$QUICK" = true ]; then
    print_warning "Quick mode: Skipping mypy and bandit"
    CMD="$CMD --hook-stage manual"
    SKIP="SKIP=mypy,bandit $CMD"
else
    SKIP="$CMD"
fi

# Run checks
echo ""
print_header "Running Checks..."
echo ""

if $SKIP; then
    echo ""
    print_success "All checks passed! 🎉"
    echo ""

    if [ "$QUICK" = true ]; then
        print_warning "Quick mode was used. Run full checks before committing:"
        echo "  ./scripts/run-checks.sh"
    fi

    exit 0
else
    EXIT_CODE=$?
    echo ""
    print_error "Some checks failed!"
    echo ""

    if [ "$FIX" = true ]; then
        print_header "🔧 Attempting Auto-fix..."

        # Try to auto-fix with ruff
        if uv run ruff check --fix .; then
            print_success "Ruff auto-fix completed"
        fi

        if uv run ruff format .; then
            print_success "Ruff format completed"
        fi

        echo ""
        print_warning "Some issues were auto-fixed. Please review changes and re-run checks."
        echo "  git diff"
        echo "  ./scripts/run-checks.sh"
    else
        echo "💡 Tips:"
        echo "  • Run with --fix to auto-fix some issues: ./scripts/run-checks.sh --fix"
        echo "  • Check specific hook: pre-commit run <hook-name> --all-files"
        echo "  • Show diff on failure: pre-commit run --all-files --show-diff-on-failure"
        echo ""
        echo "Available hooks:"
        echo "  • ruff           - Python linting"
        echo "  • ruff-format    - Python formatting"
        echo "  • mypy           - Type checking"
        echo "  • bandit         - Security scan"
    fi

    exit $EXIT_CODE
fi
