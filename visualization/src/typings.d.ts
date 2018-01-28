declare module 'react-visjs-timeline';
declare module 'd3-v4-grid';
declare module 'redux-persist';
declare module 'redux-persist/*';
declare module '*.scss';

interface WebpackWorker extends Worker {
    new(): WebpackWorker;
}

interface FileReaderEventTarget extends EventTarget
{
    result: string;
}

interface FileReaderEvent extends Event
{
    target: FileReaderEventTarget;
    getMessage(): string;
}
