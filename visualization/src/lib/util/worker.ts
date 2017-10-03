import {Observable} from 'rxjs/Observable';
import {Subject} from 'rxjs/Subject';

/**
 * Creates a worker job with a single input and output.
 * @param {string} file File with code for the worker.
 * @param {Input} data Input data.
 * @returns {Observable<Output>} Output data.
 */
export function createWorkerJob<Input, Output>(file: string, data: Input): Observable<Output>
{
    const subject = new Subject<Output>();
    const worker = new Worker(file);

    worker.onmessage = message =>
    {
        subject.next(message.data);
        subject.complete();
        worker.terminate();
    };
    worker.onerror = error =>
    {
        subject.error(error);
        worker.terminate();
    };
    worker.postMessage(data);

    return subject;
}
