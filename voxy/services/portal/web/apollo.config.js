module.exports = {
  client: {
    includes: ["src/**/*.ts"],
    service: {
      name: "voxel-api",
      localSchemaFile: "../../lib/graphql/schema.graphql",
    },
    tagName: "gql",
  },
};
