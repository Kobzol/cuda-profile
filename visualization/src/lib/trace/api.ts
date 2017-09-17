import {Observable} from "rxjs/Observable";
import {Trace} from "../data/trace";
import {readFileText} from "../util/fs";

export function parseTraceFileJson(file: File): Observable<Trace>
{
    return readFileText(file).map(data => JSON.parse(data));
}
