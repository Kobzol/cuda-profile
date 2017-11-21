import React, {PureComponent} from 'react';
import {Kernel} from '../../../lib/profile/kernel';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';

import './warp-panel.scss';

interface Props
{
    kernel: Kernel;
    trace: Trace;
    selectedWraps: Warp[];
    selectWarps: (warps: Warp[]) => void;
}

export class WarpPanel extends PureComponent<Props>
{
    render()
    {
        const warps = [...this.props.trace.warps];
        warps.sort((a: Warp, b: Warp) => {
            if (a.timestamp === b.timestamp) return 0;
            return a.timestamp < b.timestamp ? -1 : 1;
        });

        console.log(warps.length);

        const start = new Date(this.props.trace.start);
        const end = new Date(this.props.trace.end);

        return <div></div>;
    }
}
