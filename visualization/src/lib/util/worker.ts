import {Subject} from "rxjs/Subject";
import {Observable} from "rxjs/Observable";

const workerCache: {[key: string]: Worker} = {};

function getWorker(file: string, cache: boolean): Worker
{
    if (cache && workerCache.hasOwnProperty(file)) return workerCache[file];

    const worker = new Worker(file);
    if (cache) workerCache[file] = worker;
    return worker;
}

export function createWorkerJob<Input, Output>(file: string, data: Input, cache: boolean = false): Observable<Output>
{
    const subject = new Subject<Output>();
    const worker = getWorker(file, cache);

    worker.onmessage = message =>
    {
        subject.next(message.data);
        subject.complete();

        if (!cache)
        {
            worker.terminate();
        }
    };
    worker.onerror = error =>
    {
        subject.error(error);

        if (!cache)
        {
            worker.terminate();
        }
    };
    worker.postMessage(data);

    return subject;
}
