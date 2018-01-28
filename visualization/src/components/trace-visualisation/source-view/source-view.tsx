import React, {PureComponent} from 'react';
import {Warp} from '../../../lib/profile/warp';
import * as brace from 'brace';
import AceEditor from 'react-ace';
import 'brace/mode/c_cpp';
import 'brace/theme/chrome';
import * as _ from 'lodash';
import {SourceLocation} from '../../../lib/profile/metadata';
import {Button, Glyphicon, Panel} from 'react-bootstrap';

import style from './source-view.scss';

interface Props
{
    content: string;
    file: string;
    warps: Warp[];
    locationFilter: SourceLocation[];
    setLocationFilter: (line: SourceLocation[]) => void;
    onClose: () => void;
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
        return (
            <Panel header={this.renderHeader()} className={style.sourceWrapper}>
                <AceEditor
                    mode='c_cpp'
                    theme='chrome'
                    width='490px'
                    readOnly={true}
                    onLoad={this.onLoad}
                    value={this.props.content} />
            </Panel>
        );
    }
    renderHeader = (): JSX.Element =>
    {
        return (
            <div className={style.sourceHeader}>
                <div>{this.props.file}</div>
                <Button onClick={this.props.onClose} title='Close'>
                    <Glyphicon glyph='remove' />
                </Button>
            </div>
        );
    }

    onLoad = (ace: brace.Editor) =>
    {
        this.ace = ace;

        interface Event
        {
            getDocumentPosition(): { row: number };
        }

        const lineMap = _.groupBy(this.props.warps, (warp: Warp) => warp.location.line);
        this.ace.on('guttermousedown', (event: Event) => {
            const line = event.getDocumentPosition().row + 1;
            if (lineMap.hasOwnProperty(line))
            {
                const location = {
                    line,
                    file: this.props.file
                };

                const sameLoc = this.props.locationFilter.find(filter => _.isEqual(filter, location));
                if (sameLoc !== undefined)
                {
                    this.props.setLocationFilter(_.without(this.props.locationFilter, sameLoc));
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
        const lineMap = _.groupBy(this.props.warps, (warp: Warp) => warp.location.line);
        this.ace.session.setAnnotations(Object.keys(lineMap).map(line => ({
            row: parseInt(line) - 1,
            column: 0,
            type: 'warning',
            text: `${lineMap[line].length} warps`
        })));
    }
}
