#!/bin/bash
set -e

# ECM Manager Agent - Deploy Script
# Usage: ./deploy.sh [k8s|ecs|local]
# Docker context is repo root: cd manager/deploy && docker build -f Dockerfile ../..

TARGET=${1:-local}
IMAGE_TAG=${2:-latest}
REGISTRY=${ECR_REPO:-ghcr.io/vance-club}
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Deploying ECM Manager Agent"
echo "Target: $TARGET"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Build Docker image (context = repo root so shared/ is accessible)
build() {
    echo "Building Docker image..."
    docker build -t ecm-manager:$IMAGE_TAG -f "$REPO_ROOT/manager/deploy/Dockerfile" "$REPO_ROOT"
    docker tag ecm-manager:$IMAGE_TAG $REGISTRY/ecm-manager:$IMAGE_TAG
    echo "Built: $REGISTRY/ecm-manager:$IMAGE_TAG"
}

# Push to registry
push() {
    echo "Pushing to registry..."
    docker push $REGISTRY/ecm-manager:$IMAGE_TAG
    echo "Pushed"
}

# Deploy to Kubernetes
deploy_k8s() {
    echo "Deploying to Kubernetes..."
    kubectl apply -f "$REPO_ROOT/manager/deploy/k8s/cronjob.yaml"

    echo "Creating secrets..."
    kubectl create secret generic ecm-triage-secrets \
        --namespace=ecm-operations \
        --from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
        --from-literal=SLACK_BOT_TOKEN="$SLACK_BOT_TOKEN" \
        --dry-run=client -o yaml | kubectl apply -f -

    echo "Deployed to K8s"
    kubectl get cronjob ecm-manager -n ecm-operations
}

# Deploy to ECS
deploy_ecs() {
    echo "Deploying to ECS..."
    envsubst < "$REPO_ROOT/manager/deploy/ecs/task-definition.json" > /tmp/task-def.json
    aws ecs register-task-definition --cli-input-json file:///tmp/task-def.json
    echo "Deployed to ECS"
}

# Run locally
run_local() {
    echo "Running locally..."

    if [ -f "$REPO_ROOT/manager/.env" ]; then
        export $(grep -v '^#' "$REPO_ROOT/manager/.env" | xargs)
    fi

    docker run --rm \
        -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
        -e SLACK_BOT_TOKEN="$SLACK_BOT_TOKEN" \
        -e SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" \
        -e SPREADSHEET_ID="$SPREADSHEET_ID" \
        ecm-manager:$IMAGE_TAG triage
}

# Test mode
test_agent() {
    echo "Testing agent..."
    docker run --rm \
        -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
        -e SLACK_BOT_TOKEN="$SLACK_BOT_TOKEN" \
        ecm-manager:$IMAGE_TAG test
}

# Main
case $TARGET in
    k8s)
        build
        push
        deploy_k8s
        ;;
    ecs)
        build
        push
        deploy_ecs
        ;;
    local)
        build
        run_local
        ;;
    test)
        build
        test_agent
        ;;
    build)
        build
        ;;
    *)
        echo "Usage: ./deploy.sh [k8s|ecs|local|test|build]"
        exit 1
        ;;
esac

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Done!"
