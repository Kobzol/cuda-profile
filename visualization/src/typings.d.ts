declare module 'react-visjs-timeline';
declare module 'd3-v4-grid';
declare module 'redux-persist';
declare module 'redux-persist/*';

interface FileReaderEventTarget extends EventTarget
{
    result: string;
}

interface FileReaderEvent extends Event
{
    target: FileReaderEventTarget;
    getMessage(): string;
}
