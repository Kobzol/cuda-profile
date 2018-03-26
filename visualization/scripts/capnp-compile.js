const { exec } = require('child_process');

const directory = '../collection/runtime/format/capnp';

exec(`capnpc --src-prefix=${directory} -o ts:src/capnp ${directory}/cupr.capnp`,
    (err, stdout, stderr) =>
    {
    if (err)
    {
        return;
    }

    console.log('Compiled Cap\'n proto schema');
});
