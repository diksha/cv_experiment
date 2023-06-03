#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#


class DbRouter:
    """
    Documentation taken from https://docs.djangoproject.com/en/3.2/topics/db/multi-db/#database-routers

    Router to route db operation to correct database/models.

    A router doesn’t have to provide all these methods – it may omit one or more of them. If one of the methods is omitted, Django will skip that router when performing the relevant check.

    Hints
    The hints received by the database router can be used to decide which database should receive a given request.
    At present, the only hint that will be provided is instance, an object instance that is related to the read or write operation that is underway. This might be the instance that is being saved, or it might be an instance that is being added in a many-to-many relation. In some cases, no instance hint will be provided at all. The router checks for the existence of an instance hint, and determine if that hint should be used to alter routing behavior.
    """

    def db_for_read(self, model, **hints):
        """
        # noqa: DAR101
        # noqa: DAR201
        Suggest the database that should be used for read operations for objects of type model.
        If a database operation is able to provide any additional information that might assist in selecting a database, it will be provided in the hints dictionary.
        Returns None if there is no suggestion.
        """
        if model._meta.app_label == "state":
            return "state"
        return "default"

    def db_for_write(self, model, **hints):
        """
        # noqa: DAR101
        # noqa: DAR201
        Suggest the database that should be used for writes of objects of type Model.
        If a database operation is able to provide any additional information that might assist in selecting a database, it will be provided in the hints dictionary.
        Returns None if there is no suggestion.
        """
        if model._meta.app_label == "state":
            return "state"
        return "default"

    def allow_relation(self, obj1, obj2, **hints):
        """
        # noqa: DAR101
        # noqa: DAR201
        Return True if a relation between obj1 and obj2 should be allowed, False if the relation should be prevented, or None if the router has no opinion. This is purely a validation operation, used by foreign key and many to many operations to determine if a relation should be allowed between two objects.
        If no router has an opinion (i.e. all routers return None), only relations within the same database are allowed.
        """
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        # noqa: DAR101
        # noqa: DAR201
        Determine if the migration operation is allowed to run on the database with alias db.
        Return True if the operation should run, False if it shouldn’t run, or None if the router has no opinion.

        The app_label positional argument is the label of the application being migrated.
        model_name is set by most migration operations to the value of model._meta.model_name (the lowercased version of the model __name__) of the model being migrated. Its value is None for the RunPython and RunSQL operations unless they provide it using hints.
        hints are used by certain operations to communicate additional information to the router.
        When model_name is set, hints normally contains the model class under the key 'model'. Note that it may be a historical model, and thus not have any custom attributes, methods, or managers. You should only rely on _meta.
        This method can also be used to determine the availability of a model on a given database.
        makemigrations always creates migrations for model changes, but if allow_migrate() returns False, any migration operations for the model_name will be silently skipped when running migrate on the db. Changing the behavior of allow_migrate() for models that already have migrations may result in broken foreign keys, extra tables, or missing tables. When makemigrations verifies the migration history, it skips databases where no app is allowed to migrate.
        """
        if db == "state":
            return app_label == "state"
        elif db == "default":
            return app_label != "state"
        return True
