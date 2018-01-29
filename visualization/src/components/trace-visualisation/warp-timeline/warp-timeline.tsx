import React, {PureComponent} from 'react';
import {Panel} from 'react-bootstrap';
import Timeline from 'react-visjs-timeline';
import {Kernel} from '../../../lib/profile/kernel';
import {Trace} from '../../../lib/profile/trace';
import {AccessType, Warp} from '../../../lib/profile/warp';
import bigInt from 'big-integer';

interface Props
{
    kernel: Kernel;
    trace: Trace;
    selectedWarps: Warp[];
    selectWarps: (warps: Warp[]) => void;
}

export class WarpTimeline extends PureComponent<Props>
{

    render()
    {
        const start = new Date(this.props.trace.start);
        const end = new Date(this.props.trace.end);

        const options = {
            width: '100%',
            minHeight: '100px',
            zoomMin: 10,
            zoomMax: 60000,
            format: {
                minorLabels: {
                    millisecond: 's.SS'
                }
            },
            showTooltips: true,
            multiselect: true,
            start, end,
            min: start,
            max: end,
            dataAttributes: ['id']
        };

        return (
            <Panel className='trace' header={`Access timeline (select access)`} bsStyle='success'>
                <Timeline
                    options={options}
                    items={this.createTimelineItems(this.props.kernel, this.props.trace)}
                    selectHandler={this.handleAccessSelect}
                    selection={this.props.selectedWarps.map(warp => warp.index)} />
            </Panel>
        );
    }

    createTimelineItems = (kernel: Kernel, trace: Trace) =>
    {
        return trace.warps.map((warp, index) => {
            const start = this.recalculateTime(trace, warp.timestamp);
            const end = this.recalculateTime(trace, warp.timestamp) + 2;
            const content = `#${index}`;
            const location = warp.location;
            const type = warp.type;
            let title = `${content}: ${warp.accessType === AccessType.Read ? 'Read' : 'Write'}`;
            title += ` ${warp.size} bytes of ${type} (${warp.accesses.length} threads)`;
            if (location !== null)
            {
                title += ` (${location.file}:${location.line})`;
            }

            return {
                id: warp.index,
                start, end,
                content, title
            };
        });
    }

    handleAccessSelect = ({items}: {items: number[]}) =>
    {
        this.props.selectWarps(items.map(index => this.props.trace.warps[index]));
    }

    private recalculateTime(trace: Trace, timestamp: string): number
    {
        const lastAccess = trace.warps[trace.warps.length - 1].timestamp;
        const firstAccess = trace.warps[0].timestamp;
        if (firstAccess === lastAccess) return (trace.start + trace.end) / 2;

        const accessLength = bigInt(lastAccess).minus(bigInt(firstAccess));
        const fraction = (bigInt(timestamp).minus(bigInt(firstAccess))).divide(accessLength);

        return trace.start + (trace.end - trace.start) * fraction.toJSNumber();
    }
}
