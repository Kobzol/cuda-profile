import {Observable} from 'rxjs/Observable';
import {Subject} from 'rxjs/Subject';
import pako from 'pako';

/**
 * Reads file from the disk and returns its content.
 * @param {File} file
 * @param {boolean} binary Whether the file should be loaded as text (string) or binary blob (ArrayBuffer)
 * @param {boolean} decompress Whether the file should be gzip decompressed (decompress implies binary)
 * @returns {Observable<string | ArrayBuffer>}
 */
export function readFile(file: File, binary: boolean = false, decompress: boolean = false)
: Observable<string | ArrayBuffer | Uint8Array>
{
    const subject = new Subject<string | ArrayBuffer | Uint8Array>();

    const reader = new FileReader();
    reader.onload = (event: FileReaderEvent) =>
    {
        const content = event.target.result;
        if (decompress)
        {
            const options = {
                raw: false
            };

            if (!binary)
            {
                options['to'] = 'string';
            }

            subject.next(pako.ungzip(content, options));
        }
        else subject.next(content);

        subject.complete();
    };
    reader.onerror = (event: ErrorEvent) =>
    {
        subject.error(event.error);
    };

    if (binary || decompress)
    {
        reader.readAsArrayBuffer(file);
    }
    else reader.readAsText(file);

    return subject;
}
