export interface Metadata
{
    typeMap: string[];
    locations: DebugLocation[];
}

export interface DebugLocation
{
    name: string;
    file: string;
    line: number;
}
