import {Observable} from 'rxjs/Observable';
import 'rxjs/add/operator/catch';
import 'rxjs/add/operator/do';
import 'rxjs/add/operator/map';
import {Metadata} from '../format/metadata';
import {readFileBinary, readFileText} from '../util/fs';
import {createWorkerJob} from '../util/worker';
import {InvalidFileContent, InvalidFileFormat} from './errors';
import {Run} from '../format/run';
import {Trace} from '../format/trace';

export enum FileType
{
    Trace = 0,
    Metadata = 1,
    Run = 2,
    Unknown = 3,
    Invalid = 4
}

export interface TraceFile
{
    name: string;
    loading: boolean;
    content: Trace | Metadata | Run | null;
    type: FileType;
    error: number;
}

export interface FileLoadData
{
    type: FileType;
    content: Trace | Metadata;
}

/**
 * Loads file and parses its content as JSON using a web worker.
 * @param {File} file
 * @returns {Observable<Object>}
 */
function parseFileJson(file: File): Observable<Trace | Metadata>
{
    return readFileText(file)
        .flatMap(data => createWorkerJob(process.env.PUBLIC_URL + './json.worker.js', data));
}
/**
 * Loads file and parses its content as Protobuf using a web worker.
 * @param {File} file
 * @returns {Observable<Object>}
 */
function parseFileProtobuf(file: File): Observable<Trace | Metadata>
{
    return readFileBinary(file)
        .flatMap(data => createWorkerJob(process.env.PUBLIC_URL + './protobuf.worker.js', data));
}
/**
 * Loads file as JSON or Protobuf, according to the extension (.json or .proto).
 * @param {File} file
 * @returns {Observable<Object>}
 */
function parseFile(file: File): Observable<Trace | Metadata>
{
    if (file.name.match(/\.json$/))
    {
        return parseFileJson(file);
    }
    else if (file.name.match(/\.protobuf$/))
    {
        return parseFileProtobuf(file).map(content => ({
            ...content,
            type: 'trace'
        }));
    }
    else return Observable.throw(new InvalidFileFormat());
}

/**
 * Checks validity of trace object.
 * @param {Object} content
 * @returns {Observable<Trace>}
 */
function validateTrace(content: object): boolean
{
    return (
        content['type'] === 'trace' &&
        'kernel' in content &&
        'accesses' in content
    );
}
/**
 * Checks validity of metadata object.
 * @param {Object} content
 * @returns {boolean}
 */
function validateMetadata(content: object): boolean
{
    return (
        content['type'] === 'metadata' &&
        'kernel' in content
    );
}
/**
 * Checks validity of run object.
 * @param {Object} content
 * @returns {boolean}
 */
function validateRun(content: object): boolean
{
    return (
        content['type'] === 'run' &&
        'start' in content &&
        'end' in content
    );
}
/**
 * Checks validity of file content depending on its type.
 * @param {File} file
 * @param {Object} content
 * @returns {boolean}
 */
function validateContent(file: File, content: object): boolean
{
    const type = getFileType(file);

    switch (type)
    {
        case FileType.Metadata: return validateMetadata(content);
        case FileType.Run: return validateRun(content);
        case FileType.Trace: return validateTrace(content);
        default: return false;
    }
}

/**
 * Returns file type, depending on file name.
 * @param {File} file
 * @returns {FileType}
 */
function getFileType(file: File): FileType
{
    if (file.name === 'run.json')
    {
        return FileType.Run;
    }

    const metadata = file.name.match(/.*\.metadata\..*/);
    return metadata ? FileType.Metadata : FileType.Trace;
}

/**
 * Loads file, parses it and creates appropriate data structure (trace or metadata).
 * @param {File} file
 * @returns {Observable<FileLoadData>}
 */
export function parseAndValidateFile(file: File): Observable<FileLoadData>
{
    return parseFile(file)
        .catch(error => {
            if (!(error instanceof InvalidFileFormat)) return Observable.throw(new InvalidFileContent());
            else return Observable.throw(error);
        })
        .do(content => {
            if (!validateContent(file, content))
            {
                throw new InvalidFileContent();
            }
        })
        .map(content => ({
            type: getFileType(file),
            content
        }));
}
