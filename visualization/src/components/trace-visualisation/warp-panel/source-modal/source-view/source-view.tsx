import React, {PureComponent} from 'react';
import {Warp} from '../../../../../lib/profile/warp';
import brace from 'brace';
import AceEditor from 'react-ace';
import 'brace/mode/c_cpp';
import 'brace/theme/chrome';
import {SourceLocation} from '../../../../../lib/profile/metadata';
import style from './source-view.scss';
import {groupBy, equals, without} from 'ramda';

interface Props
{
    content: string;
    file: string;
    warps: Warp[];
    locationFilter: SourceLocation[];
    setLocationFilter: (lines: SourceLocation[]) => void;
}

export class SourceView extends PureComponent<Props>
{
    private ace: brace.Editor;

    componentDidMount()
    {
        this.setLineAnnotations();
        this.setGutterDecorations();
    }
    componentDidUpdate()
    {
        this.setLineAnnotations();
        this.setGutterDecorations();
    }

    render()
    {
        return <AceEditor
                    mode='c_cpp'
                    theme='chrome'
                    width='490px'
                    readOnly
                    onLoad={this.onLoad}
                    value={this.props.content} />;
    }

    onLoad = (ace: brace.Editor) =>
    {
        this.ace = ace;

        interface Event
        {
            getDocumentPosition(): { row: number };
        }

        const lineMap = groupBy((warp: Warp) => warp.location.line.toString(), this.props.warps);
        this.ace.on('guttermousedown', (event: Event) => {
            const line = event.getDocumentPosition().row + 1;
            if (lineMap.hasOwnProperty(line))
            {
                const location = {
                    line,
                    file: this.props.file
                };

                const existingLocation = this.props.locationFilter.find(filter => equals(filter, location));
                if (existingLocation !== undefined)
                {
                    this.props.setLocationFilter(without([existingLocation], this.props.locationFilter));
                }
                else this.props.setLocationFilter(this.props.locationFilter.concat([location]));
            }
        });
    }

    setGutterDecorations = () =>
    {
        const gutterClass = style.gutterSelectedLine;
        this.props.warps.map(warp => warp.location.line).forEach(line => {
            this.ace.session.removeGutterDecoration(line - 1, gutterClass);
        });

        this.props.locationFilter
            .filter(location => location.file === this.props.file)
            .forEach(location => {
            this.ace.session.addGutterDecoration(location.line - 1, gutterClass);
        });
    }
    setLineAnnotations = () =>
    {
        const lineMap = groupBy((warp: Warp) => warp.location.line.toString(), this.props.warps);
        this.ace.session.setAnnotations(Object.keys(lineMap).map(line => ({
            row: parseInt(line) - 1,
            column: 0,
            type: 'warning',
            text: `${lineMap[line].length} warps`
        })));
    }
}
