export class InvalidWarpData extends Error
{
    constructor(message: string)
    {
        super(message);

        Object.setPrototypeOf(this, InvalidWarpData.prototype);
    }
}

export class MissingProfileData extends Error
{
    constructor(message: string)
    {
        super(message);

        Object.setPrototypeOf(this, MissingProfileData.prototype);
    }
}
