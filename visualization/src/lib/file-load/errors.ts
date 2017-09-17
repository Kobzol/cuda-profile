export class InvalidFileFormat extends Error
{
    constructor()
    {
        super();

        Object.setPrototypeOf(this, InvalidFileFormat.prototype);
    }
}

export class InvalidFileContent extends Error
{
    constructor()
    {
        super();

        Object.setPrototypeOf(this, InvalidFileFormat.prototype);
    }
}
