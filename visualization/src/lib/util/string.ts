export function getFilename(file: string): string
{
    const slash = file.lastIndexOf('/');
    if (slash === -1) return file;
    return file.substr(slash + 1);
}
