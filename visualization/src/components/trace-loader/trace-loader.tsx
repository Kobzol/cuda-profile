import React, {ChangeEvent, DragEvent, PureComponent} from 'react';
import {connect} from 'react-redux';
import {loadFile} from '../../lib/file-load/actions';
import {TraceFile} from '../../lib/file-load/file';
import {loadingFiles, validTraceFiles} from '../../lib/file-load/reducer';
import {GlobalState} from '../../lib/state/reducers';
import {Button, Glyphicon} from 'react-bootstrap';
import {buildProfile} from '../../lib/profile/actions';

interface StateProps
{
    files: TraceFile[];
    validTraceFiles: TraceFile[];
    loadingFiles: TraceFile[];
    buildError: string;
}

interface DispatchProps
{
    loadFile: (file: File) => {};
    buildProfile: (files: TraceFile[]) => {};
}

class TraceLoaderComponent extends PureComponent<StateProps & DispatchProps>
{
    render()
    {
        return (
            <div>
                <input type='file' multiple={true} onChange={this.handleTraceChange} onDrop={this.handleTraceDrop} />
                <ul>
                    {this.props.files.map(this.renderFile)}
                </ul>
                <Button
                    disabled={!this.canBuildProfile()}
                    onClick={this.buildProfile}
                    bsStyle='primary'>
                    <Glyphicon glyph='flash' /> Load trace
                </Button>
                {this.props.buildError && <div className='error'>{this.props.buildError}</div>}
            </div>
        );
    }

    renderFile = (file: TraceFile): JSX.Element =>
    {
        return (
            <li key={file.name}>
                <span>{file.name}, loading: {file.loading ? 'true' : 'false'}, error: {file.error}</span>
            </li>
        );
    }

    handleTraceChange = (event: ChangeEvent<HTMLInputElement>) =>
    {
        for (let i = 0; i < event.target.files.length; i++)
        {
            this.props.loadFile(event.target.files[i]);
        }
    }
    handleTraceDrop = (event: DragEvent<HTMLInputElement>) =>
    {
        const files = event.dataTransfer.files;
        for (let i = 0; i < files.length; i++)
        {
            this.props.loadFile(files[i]);
        }
    }

    canBuildProfile = (): boolean =>
    {
        return this.props.validTraceFiles.length > 0 && this.props.loadingFiles.length < 1;
    }
    buildProfile = () =>
    {
        this.props.buildProfile(this.props.validTraceFiles);
    }
}

export const TraceLoader = connect<StateProps, DispatchProps, {}>((state: GlobalState) => ({
    files: state.fileLoader.files,
    buildError: state.profile.buildError,
    validTraceFiles: validTraceFiles(state),
    loadingFiles: loadingFiles(state)
}), ({
    loadFile: loadFile.started,
    buildProfile: buildProfile.started
}))(TraceLoaderComponent);
