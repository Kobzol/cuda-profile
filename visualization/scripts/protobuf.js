var cli = require("protobufjs/cli/");
var pbjs = cli.pbjs;
var pbts = cli.pbts;

var bundlePath = "./src/protobuf/bundle";
var bundleJS = bundlePath + ".js";
var bundleTS = bundlePath + ".d.ts";

pbjs.main([
    "-t", "static-module",
    "-o", bundleJS,
    "-w", "es6",
    "../collection/runtime/format/protobuf/*.proto"],
    function (err)
    {
        if (err)
        {
            throw err;
        }

        pbts.main([
           bundleJS,
           "-o", bundleTS
        ], function (err)
        {
            if (err)
            {
                throw err;
            }

            console.log("Protobuf bundle successfully generated");
        });
    }
);
