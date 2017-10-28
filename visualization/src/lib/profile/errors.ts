export class InvalidWarpData extends Error
{
    constructor(message: string)
    {
        super(message);

        Object.setPrototypeOf(this, InvalidWarpData.prototype);
    }
}
