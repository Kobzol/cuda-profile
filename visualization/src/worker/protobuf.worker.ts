import {cupr} from '../protobuf/bundle';

const ctx: Worker = self as {} as Worker;

ctx.onmessage = message =>
{
    ctx.postMessage(cupr.proto.KernelTrace.decode(new Uint8Array(message.data)).toJSON());
};

export default {} as WebpackWorker;
