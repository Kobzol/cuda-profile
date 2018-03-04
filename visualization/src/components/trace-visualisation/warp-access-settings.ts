import chroma from 'chroma-js';

export const WRITE_COLOR = chroma(180, 20, 0);
export const READ_COLOR = chroma(20, 180, 20);

export function getIdentifier(index: number): string
{
    if (index >= 26) return (index - 26).toString();
    return String.fromCharCode('a'.charCodeAt(0) + index);
}
