.PHONY: codegen graphql-gen dev-kube portal-fe portal-be migrations

codegen:
	./services/portal/scripts/codegen.sh

graphql-gen:
	./core/portal/scripts/development/graphql_sync.sh

dev-kube:
	./core/scripts/local_dev.sh

portal-fe:
	./services/portal/web/scripts/devserver.sh

portal-be:
	./core/portal/scripts/development/devserver_backend.sh

migrations:
	./core/portal/scripts/makemigrations.sh
