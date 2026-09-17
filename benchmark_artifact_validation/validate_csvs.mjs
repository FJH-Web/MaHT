import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const projectRoot = path.resolve("..");
const outputDir = path.join(projectRoot, "benchmark_staging", "output");
const previewDir = path.join(projectRoot, "benchmark_staging", "validation_previews");
await fs.mkdir(previewDir, { recursive: true });

const files = [
  "NHANES_statbench.csv",
  "METABRIC_statbench.csv",
  "MIMIC_IV_demo_statbench.csv",
  "Inside_Airbnb_NYC_statbench.csv",
  "Bike_Sharing_statbench.csv",
];

const results = [];
for (const filename of files) {
  const csvText = await fs.readFile(path.join(outputDir, filename), "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Data" });
  const inspected = await workbook.inspect({
    kind: "region",
    sheetId: "Data",
    range: "A1:L6",
    maxChars: 3500,
  });
  if (!inspected.ndjson || inspected.ndjson.length === 0) {
    throw new Error(`Empty inspection result for ${filename}`);
  }
  const preview = await workbook.render({
    sheetName: "Data",
    range: "A1:L8",
    scale: 1,
    format: "png",
  });
  const previewPath = path.join(
    previewDir,
    `${path.basename(filename, ".csv")}.png`,
  );
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  results.push({
    filename,
    inspectChars: inspected.ndjson.length,
    previewPath,
  });
}

console.log(JSON.stringify(results, null, 2));
