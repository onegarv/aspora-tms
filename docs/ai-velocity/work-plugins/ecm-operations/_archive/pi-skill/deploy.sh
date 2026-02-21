#!/bin/bash
set -e

# ECM Triage Skill - Deploy Script
# Usage: ./deploy.sh [k8s|ecs|local]

TARGET=${1:-local}
IMAGE_TAG=${2:-latest}
REGISTRY=${ECR_REPO:-ghcr.io/vance-club}

echo "🚀 Deploying ECM Triage Skill"
echo "Target: $TARGET"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Build Docker image
build() {
    echo "📦 Building Docker image..."

    # Resolve symlinks - copy actual SQL files
    echo "Copying queries..."
    rm -rf queries/*.sql 2>/dev/null || true
    cp ../queries/*.sql queries/ 2>/dev/null || echo "No queries to copy"

    docker build -t ecm-triage-skill:$IMAGE_TAG .
    docker tag ecm-triage-skill:$IMAGE_TAG $REGISTRY/ecm-triage-skill:$IMAGE_TAG
    echo "✅ Built: $REGISTRY/ecm-triage-skill:$IMAGE_TAG"
}

# Push to registry
push() {
    echo "⬆️ Pushing to registry..."
    docker push $REGISTRY/ecm-triage-skill:$IMAGE_TAG
    echo "✅ Pushed"
}

# Deploy to Kubernetes
deploy_k8s() {
    echo "☸️ Deploying to Kubernetes..."

    # Create namespace
    kubectl apply -f k8s/cronjob.yaml

    # Create secrets (prompt for values)
    echo "Creating secrets..."
    kubectl create secret generic ecm-triage-secrets \
        --namespace=ecm-operations \
        --from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
        --from-literal=SLACK_BOT_TOKEN="$SLACK_BOT_TOKEN" \
        --dry-run=client -o yaml | kubectl apply -f -

    echo "✅ Deployed to K8s"
    kubectl get cronjob ecm-triage-skill -n ecm-operations
}

# Deploy to ECS
deploy_ecs() {
    echo "🐳 Deploying to ECS..."

    # Register task definition
    envsubst < ecs/task-definition.json > /tmp/task-def.json
    aws ecs register-task-definition --cli-input-json file:///tmp/task-def.json

    # Create EventBridge rule
    envsubst < ecs/eventbridge-schedule.json > /tmp/schedule.json
    aws events put-rule --cli-input-json file:///tmp/schedule.json

    echo "✅ Deployed to ECS"
}

# Run locally
run_local() {
    echo "💻 Running locally..."

    # Load env
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    fi

    # Run with Docker
    docker run --rm \
        -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
        -e SLACK_BOT_TOKEN="$SLACK_BOT_TOKEN" \
        -e SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" \
        -e SPREADSHEET_ID="$SPREADSHEET_ID" \
        ecm-triage-skill:$IMAGE_TAG triage
}

# Test mode
test_skill() {
    echo "🧪 Testing skill..."
    docker run --rm \
        -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
        -e SLACK_BOT_TOKEN="$SLACK_BOT_TOKEN" \
        ecm-triage-skill:$IMAGE_TAG test
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
        test_skill
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
echo "✅ Done!"
