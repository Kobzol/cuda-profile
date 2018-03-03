import React from 'react';
import {PureComponent} from 'react';
import {MemoryAllocation} from '../../../../../lib/profile/memory-allocation';
import {AddressRange, WarpAddressSelection} from '../../../../../lib/trace/selection';
import {getAllocationAddressRange, getSelectionRange} from '../../../../../lib/profile/address';

interface Props
{
    width: number;
    height: number;
    allocation: MemoryAllocation;
    rangeSelections: WarpAddressSelection[];
}

export class MemoryMinimap extends PureComponent<Props>
{
    private canvasRef: HTMLCanvasElement;

    componentDidMount()
    {
        this.redraw(this.canvasRef);
    }

    componentDidUpdate()
    {
        this.redraw(this.canvasRef);
    }

    render()
    {
        return (
            <canvas
                width={this.props.width}
                height={this.props.height}
                ref={canvas => this.canvasRef = canvas} />
        );
    }

    redraw = (canvas: HTMLCanvasElement) =>
    {
        const dim = 20;
        const size = this.props.allocation.size;

        const blockSize = Math.max(1, Math.ceil(size / (dim * dim)));
        const blockCount = Math.ceil(size / blockSize);
        const blockHeight = canvas.height / dim;

        const ctx = canvas.getContext('2d');
        ctx.save();

        const rows = Math.floor(size / (blockSize * dim));

        ctx.fillStyle = 'rgb(0, 0, 200)';

        if (rows > 0)
        {
            ctx.fillRect(0, 0, canvas.width, rows * blockHeight);
        }

        const remaining = blockCount - (rows * dim);
        if (remaining > 0)
        {
            ctx.fillRect(0, rows * blockHeight, canvas.width * (remaining / dim), blockHeight);
        }

        ctx.restore();
    }

    getAddressRange = (): AddressRange =>
    {
        return getSelectionRange(getAllocationAddressRange(this.props.allocation), this.props.rangeSelections);
    }
}
