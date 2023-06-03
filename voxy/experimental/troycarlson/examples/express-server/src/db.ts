import { Client } from "pg";

export const doDbStuff = async () => {
  const client = new Client({
    host: "localhost",
    database: "voxeldev",
    port: 31003,
    user: "voxelapp",
    password: "voxelvoxel",
  });

  await client.connect();

  const res = await client.query("SELECT $1::text as message", ["Hello world!"]);
  console.log(res.rows[0].message); // Hello world!
  await client.end();
};
