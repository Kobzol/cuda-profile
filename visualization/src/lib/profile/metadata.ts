export interface Metadata
{
    typeMap: string[];
    nameMap: string[];
    locations: DebugLocation[];
    source: SourceLocation & { content: string };
}

export interface DebugLocation
{
    name: string;
    file: string;
    line: number;
}

export interface SourceLocation
{
    file: string;
    line: number;
}
