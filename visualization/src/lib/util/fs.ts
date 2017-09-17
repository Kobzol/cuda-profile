import {Observable} from "rxjs/Observable";
import {Subject} from "rxjs/Subject";

export function readFile(file: File, binary: boolean = false): Observable<string | ArrayBuffer>
{
    const subject = new Subject<string>();

    const reader = new FileReader();
    reader.onload = (event: any) =>
    {
        subject.next(event.target.result);
        subject.complete();
    };
    reader.onerror = (event: ErrorEvent) =>
    {
        subject.error(event.error);
    };

    if (binary)
    {
        reader.readAsArrayBuffer(file);
    }
    else reader.readAsText(file);

    return subject;
}

export function readFileText(file: File): Observable<string>
{
    return readFile(file) as Observable<string>;
}
export function readFileBinary(file: File): Observable<ArrayBuffer>
{
    return readFile(file, true) as Observable<ArrayBuffer>;
}
