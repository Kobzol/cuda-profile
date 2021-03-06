import {InvalidFileContent, InvalidFileFormat} from '../file-load/errors';

export const Errors = {
    None: 0,
    Unknown: 1,

    InvalidFileFormat: 10,
    InvalidFileContent: 11
};

export const getErrorId = (error: Error) =>
{
    if (error instanceof InvalidFileFormat) return Errors.InvalidFileFormat;
    if (error instanceof InvalidFileContent) return Errors.InvalidFileContent;

    console.error(error);
    return Errors.Unknown;
};
