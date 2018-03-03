const ctx: Worker = self as {} as Worker;

ctx.onmessage = message =>
{
    ctx.postMessage(JSON.parse(message.data));
};

export default {} as WebpackWorker;
