import {Observable} from "rxjs/Observable";
import {Subject} from "rxjs/Subject";

export function readFileText(file: File): Observable<string>
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

    reader.readAsText(file);

    return subject;
}
