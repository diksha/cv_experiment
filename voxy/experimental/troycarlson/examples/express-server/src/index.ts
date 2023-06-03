import { Express, Request, Response } from "express";
import * as express from "express";
import { bar } from "./foo";
import { doDbStuff } from "./db";

const app: Express = express();
const port = 8000;

app.get("/", async (req: Request, res: Response) => {
  await doDbStuff();
  bar("bazz");
  res.send("Hello World!");
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
