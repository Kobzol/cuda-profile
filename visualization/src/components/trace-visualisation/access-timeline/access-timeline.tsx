import React, {PureComponent} from 'react';
import {Panel} from 'react-bootstrap';
import Timeline from 'react-visjs-timeline';
import {Kernel} from '../../lib/profile/kernel';
import {Trace} from '../../lib/profile/trace';
import {AccessType} from '../../lib/profile/memory-access';

interface Props
{
    kernel: Kernel;
    trace: Trace;
    selectAccessGroup: (index: number) => void;
}

export class AccessTimeline extends PureComponent<Props>
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
            start, end,
            min: start,
            max: end,
            dataAttributes: ['id']
        };
        return (
            <Panel className='trace' header={`Access timeline (select access)`}>
                <Timeline
                    options={options}
                    items={this.createTimelineItems(this.props.kernel, this.props.trace)}
                    selectHandler={this.handleAccessSelect}
                />
            </Panel>
        );
    }

    createTimelineItems = (kernel: Kernel, trace: Trace) =>
    {
        return trace.accessGroups.map((group, index) => {
            const start = this.recalculateTime(trace, group.timestamp);
            const end = this.recalculateTime(trace, group.timestamp) + 2;
            const content = `#${index}`;
            const location = group.debugId !== 1 ? kernel.metadata.locations[group.debugId] : null;
            const type = kernel.metadata.typeMap[group.typeIndex];
            let title = `${content}: ${group.kind === AccessType.Read ? 'Read' : 'Write'}`;
            title += ` ${group.size} bytes of ${type} (${group.accesses.length} threads)`;
            if (location !== null)
            {
                title += ` (${location.file}:${location.line})`;
            }

            return {
                id: `${index}`,
                start, end,
                content, title
            };
        });
    }

    handleAccessSelect = ({items}: {items: string[]}) =>
    {
        this.props.selectAccessGroup(parseInt(items[0], 10));
    }

    private recalculateTime(trace: Trace, timestamp: number): number
    {
        const lastAccess = trace.accessGroups[trace.accessGroups.length - 1].timestamp;
        const firstAccess = trace.accessGroups[0].timestamp;
        if (firstAccess === lastAccess) return (trace.start + trace.end) / 2;

        const accessLength = lastAccess - firstAccess;
        const fraction = (timestamp - firstAccess) / accessLength;

        return trace.start + (trace.end - trace.start) * fraction;
    }
}
