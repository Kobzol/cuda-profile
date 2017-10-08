export interface Metadata
{
    type: string;
    kernel: string;
    typeMap: string[];
    locations: DebugLocation[];
}

export interface DebugLocation
{
    name: string;
    file: string;
    line: number;
}
