importScripts('http://cdn.rawgit.com/dcodeIO/protobuf.js/6.8.0/dist/protobuf.min.js');
importScripts('./proto-bundle.js');

self.onmessage = function(message)
{
    self.postMessage(protobuf.roots.default.KernelTrace.decode(new Uint8Array(message.data)).toJSON());
};
