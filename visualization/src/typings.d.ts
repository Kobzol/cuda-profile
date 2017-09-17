declare module 'react-fabricjs';

interface FileReaderEventTarget extends EventTarget
{
    result: string;
}

interface FileReaderEvent extends Event
{
    target: FileReaderEventTarget;
    getMessage(): string;
}
