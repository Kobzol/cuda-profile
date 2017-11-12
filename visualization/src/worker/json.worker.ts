const ctx: Worker = self as any;

ctx.onmessage = message =>
{
    ctx.postMessage(JSON.parse(message.data));
};

export default {} as WebpackWorker;
