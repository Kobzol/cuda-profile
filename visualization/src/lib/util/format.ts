export function formatMiB(size: number): string
{
    const kb = size / 1024;
    const mb = kb / 1024;
    return mb.toString(10);
}
