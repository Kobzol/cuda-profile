import React, {ChangeEvent, DragEvent, PureComponent} from 'react';
import {connect} from 'react-redux';
import {loadFile} from '../../lib/file-load/actions';
import {FileType, TraceFile} from '../../lib/file-load/file';
import {loadingFiles, validTraceFiles} from '../../lib/file-load/reducer';
import {GlobalState} from '../../lib/state/reducers';
import {Button, Glyphicon, Label, ListGroup, ListGroupItem, Panel, ProgressBar} from 'react-bootstrap';
import {buildProfile} from '../../lib/profile/actions';
import {Errors} from '../../lib/state/errors';
import {withRouter} from 'react-router';

import style from './trace-loader.scss';

interface StateProps
{
    files: TraceFile[];
    validTraceFiles: TraceFile[];
    loadingFiles: TraceFile[];
    buildError: string;
}

interface DispatchProps
{
    loadFile: (file: File) => void;
    buildProfile: (files: TraceFile[]) => void;
}

class TraceLoaderComponent extends PureComponent<StateProps & DispatchProps>
{
    render()
    {
        return (
            <div>
                <Panel header='Select recorded trace files' bsStyle='primary'>
                    <input className={style.fileInput}
                           type='file' multiple
                           onChange={this.handleTraceChange}
                           onDrop={this.handleTraceDrop} />
                    <ListGroup>
                        {this.props.files.map(this.renderFile)}
                    </ListGroup>
                    <Button
                        disabled={!this.canBuildProfile()}
                        onClick={this.buildProfile}
                        bsStyle='primary'>
                        <Glyphicon glyph='flash' /> Show trace
                    </Button>
                    {this.props.buildError && <div className='error'>{this.props.buildError}</div>}
                </Panel>
            </div>
        );
    }

    renderFile = (file: TraceFile): JSX.Element =>
    {
        return (
            <ListGroupItem key={file.name} className={style.file}>
                <span className={style.fileLabel}>{this.renderFileLabel(file)}</span>
                <span className={style.fileName}>{file.name}</span>
            </ListGroupItem>
        );
    }
    renderFileLabel = (file: TraceFile): JSX.Element =>
    {
        if (file.loading) return this.renderFileProgress();
        if (file.error === Errors.None) return this.renderFileSuccess(file);
        return this.renderFileError(file);
    }
    renderFileProgress = (): JSX.Element =>
    {
        return <ProgressBar striped active now={100} className={style.fileProgress} />;
    }
    renderFileError = (file: TraceFile): JSX.Element =>
    {
        return (
            <Label bsStyle='danger' className={style.label}>
                {this.formatError(file.error)}
            </Label>
        );
    }
    renderFileSuccess = (file: TraceFile): JSX.Element =>
    {
        return (
            <Label bsStyle='success' className={style.label}>
                Loaded ({this.formatFileType(file.type)})
            </Label>
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

    formatError = (error: number): string =>
    {
        switch (error)
        {
            case Errors.InvalidFileContent: return 'Invalid file content';
            case Errors.InvalidFileFormat: return 'Invalid file format';
            default: return 'Unknown error';
        }
    }
    formatFileType = (type: FileType): string =>
    {
        switch (type)
        {
            case FileType.Metadata: return 'metadata file';
            case FileType.Run: return 'run file';
            case FileType.Trace: return 'trace file';
            default: return 'unknown file';
        }
    }
}

export const TraceLoader = withRouter(connect<StateProps, DispatchProps>((state: GlobalState) => ({
    files: state.fileLoader.files,
    buildError: state.profile.buildError,
    validTraceFiles: validTraceFiles(state),
    loadingFiles: loadingFiles(state)
}), ({
    loadFile: loadFile.started,
    buildProfile: buildProfile.started
}))(TraceLoaderComponent));
