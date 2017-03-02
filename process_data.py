#!/usr/bin/python3
#
# Copyright (c) 2014 Matthias Klumpp <matthias@tenstral.net>
#
# Licensed under the GNU Lesser General Public License Version 2.1+
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 2.1 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import csv
from pylab import *
from operator import itemgetter
from scipy.interpolate import interp1d
from scipy.integrate import simps, trapz
import yaml
from PyQt4 import QtGui

__version__ = '1.4'

def debugln(msg):
    print("[DEBUG] %s" % (str(msg)))

class GCaMPData():
    def __init__(self, csv_fname):
        self.csv_fname = csv_fname

        self.timeline  = list()
        self.raw_data = list()

        self.slicedata = list()
        self.slicedata.append({'index': 0, 'name': "Main", 'frequency': -1, 'percentage': -1, 'individual_normalize': False})

        self.regression_line = list()
        self.normalized = list()
        self._events = list()
        self._filtered_curve = list()

        # true if slices should be normalized individually
        self.individual_normalize = False

        # path to our data
        self.data_dir = os.path.dirname(self.csv_fname)

        # automatically extract the name for this dataset
        basepath_noext = os.path.splitext(csv_fname)[0]
        basename_noext = os.path.basename(basepath_noext)
        if (basename_noext.startswith('data-')):
            basename_noext = basename_noext.replace('data-', '', 1)
        self.name = basename_noext

    def is_valid(self):
        if (len(self.timeline) == 0) or (len(self.raw_data) == 0):
            return False
        return True

    def has_application_data(self):
        return len(self.slicedata) > 1

    def reset_slices(self):
        self.slicedata = list()

    def update_slice_index(self):
        # correct the index of the main slice, in case we didn't do applications
        if not self.has_application_data():
            if self.events:
                self.slicedata[0]['index'] = len(self.events)
            else:
                self.slicedata[0]['index'] = len(self.raw_data)

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, value):
        self._events = value
        self.update_slice_index()

    @property
    def filtered_curve(self):
        return self._filtered_curve

    @filtered_curve.setter
    def filtered_curve(self, value):
        self._filtered_curve = value
        self.update_slice_index()

class GCaMPDataProcessor():
    processed_data = list()

    def __init__(self):
        pass

    def _read_hints_data(self, gdata):
        hintdata = None
        # check if we have annotation about substance application
        hints_fname = os.path.join(os.path.dirname(gdata.csv_fname), "%s.hints.yml" % (gdata.name))
        if os.path.isfile(hints_fname):
            debugln("Found hints for '%s'" % (gdata.name))
            with open(hints_fname, 'rt') as ymlfile:
                hintdata = yaml.safe_load(ymlfile)
            if hintdata.get('IndividualNormalize'):
                gdata.individual_normalize = True
        return hintdata

    def _find_time_index(self, time, timeline):
        if time == 0:
            return 0
        for t in timeline:
            if t >= time:
                return timeline.index(t)
        return None

    def _get_slice_data(self, hdata):
        if not hdata:
            return False
        slices = hdata.get("Slices")
        if len(slices) == 0:
            return False
        return slices

    def _rewrite_timeline_applications(self, gdata, hints):
        """ This function 'applies the application times' by rewriting the original input data
        and application times for future use, erasing periods where substances were applied.
        The function stores application time cutoffs in GCaMPData """
        slices = self._get_slice_data(hints)
        if not slices:
            return False

        # we don't trust users to have written "sane" annotation files, so sort that stuff again
        slices.sort(key = itemgetter('start'))
        gdata.reset_slices()

        time_removed = 0
        prev_endtime = 0
        tmp_data = list()
        for atime in slices:
            if gdata.individual_normalize:
                indiv_normalize = True
            else:
                indiv_normalize = atime.get('individualNormalize', False)
            starttime_val = atime['start']
            endtime_val = atime['end']
            if (endtime_val == "END") or (endtime_val < 0):
                endtime_val = gdata.timeline[-1]

            # find correct point in time (storing indices on the timeline)
            starttime = self._find_time_index(starttime_val, gdata.timeline)
            endtime = self._find_time_index(endtime_val, gdata.timeline)
            if (starttime == None) or (not endtime):
                print("Bogus application data information for '%s': Exceeds limits. (Dataset: %s)" % (gdata.name, atime))
                continue

            tmp_data.extend(gdata.raw_data[starttime:endtime])

            # add index hint
            time_removed = time_removed + (starttime - prev_endtime)
            gdata.slicedata.append({'index': (endtime - time_removed), 'name': atime['name'], 'frequency': -1, 'percentage': -1, 'individual_normalize': indiv_normalize})
            prev_endtime = endtime

        # rewrite timeline and dataset
        if len(tmp_data) <= 0:
            # no slice was added, probably due to bogus annotation file
            # try to rescue the analysis
            gdata.slicedata.append({'index': 0, 'name': "Main", 'frequency': -1, 'percentage': -1, 'individual_normalize': False})
            return False
        gdata.raw_data = tmp_data
        gdata.timeline = gdata.timeline[:len(gdata.raw_data)]

        return True

    def _count_events(self, events, timeline):
        """ Count the number of events to get activity in Hz """
        events_count = 0
        for v in events:
            if v > 0:
                events_count += 1
        activity_hz = events_count / timeline[-1]
        return real(activity_hz)

    def _calculate_activity_percentage(self, filtered_curve, timeline):
        """ Calculate activity percentage using integration of data """
        area_active = simps(filtered_curve, timeline)
        area_possible = amax(filtered_curve)*amax(timeline)
        if (area_possible == 0) or (area_active == 0):
            activity_percentage = 0
        else:
            activity_percentage = 100/area_possible*area_active
        return real(activity_percentage)

    def _figure_set_xstepsize(self, ax, stepsize):
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, stepsize))

    def _plot_data(self, ax, gdata, data, annotate=False, print_frequency=False):
        colors = iter(cm.rainbow(np.linspace(0, 1.5, len(gdata.slicedata)+8)))
        prev_index = 0
        if len(gdata.slicedata) == 1:
            # if there is only one slice, we want a nice color (blue is good!)
            next(colors)

        for ad in gdata.slicedata:
            idx = ad['index']
            color = next(colors)
            ax.plot(gdata.timeline[prev_index:idx],data[prev_index:idx], color=color)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(10)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(10)
            if annotate:
                if print_frequency:
                    text = "Frequency: %.2f Hz" % (round(float(ad['frequency']), 2))
                else:
                    text = "Activity: %.2f%%" % (round(float(ad['percentage']), 2))
                    ax.fill_between(gdata.timeline[prev_index:idx],data[prev_index:idx], color='#ffd9a8')
                ax.annotate(text, xy=(gdata.timeline[prev_index],
                                    amax(data)),
                                    xytext=(gdata.timeline[prev_index]+sqrt(amax(gdata.timeline))/4,
                                    amax(data)-sqrt(amax(data))/12),
                                    size=8,
                                    color=color)
            prev_index = idx

    def process_csv(self, csv_fname):
        # we ignore result files, which accidentially might land here
        if csv_fname.endswith("-result.csv"):
            return None

        gdata = GCaMPData(csv_fname)
        with open(gdata.csv_fname, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for row in reader:
                try:
                    gdata.timeline.append(float(row[0]))
                    gdata.raw_data.append(float(row[1]))
                except:
                    # it's highly likely that we've hit the header here
                    if row[0] != "Time":
                        print("Ignoring row: %s" % (str(row)))
        if not gdata.is_valid():
            print("Ignoring file '%s': Data looks broken." % (gdata.csv_fname))
            return None

        original_raw_data = gdata.raw_data[:]

        hints = self._read_hints_data(gdata)
        # in case substances were applied, we need a cleaned timeline
        if not self._rewrite_timeline_applications(gdata, hints):
            original_raw_data = None

        # hack to keep single-slice data in sync
        gdata.update_slice_index()

        # maybe we need to normalize the items individually...
        gdata.regression_line = list()
        prev_index = 0
        for sl in gdata.slicedata:
            idx = sl['index']
            if sl.get('individual_normalize', False):
                data = gdata.raw_data[prev_index:idx]
                m_timeline = gdata.timeline[:len(data)]
                (ar,br) = polyfit(m_timeline, data, 1)
                gdata.regression_line.extend(polyval([ar,br], m_timeline))
            else:
                (ar,br) = polyfit(gdata.timeline, gdata.raw_data, 1)
                regl = polyval([ar,br], gdata.timeline)
                gdata.regression_line.extend (regl[prev_index:idx])
            prev_index = idx

        gdata.normalized = list()
        s_curve = list()
        prev_val = gdata.regression_line[0]
        for i in range(len(gdata.raw_data)):
            val = gdata.raw_data[i]
            regr = gdata.regression_line[i]
            if (val < regr) or (val <= prev_val):
                gdata.normalized.append(0)
            else:
                gdata.normalized.append(val - regr)
            s_curve.append(val - regr)
            prev_val = val

        gdata.normalized = array(gdata.normalized)
        s_curve = array(s_curve)

        # NOTE: Disabled - maybe FFT will be useful later
        # take the fourier transformation of the data
        #F = fft(gdata.normalized)
        #bp=F[:]

        #for i in range(len(bp)):
        #    if i>=100:
        #        bp[i]=0

        #gdata.fourier=ifft(bp)

        # try to guess a good cutoff threshold per slice
        gdata.events = list()
        prev_index = 0
        for sl in gdata.slicedata:
            s_tmp = list()
            idx = sl['index']
            idx_end = idx
            idx_start = prev_index
            cutoff_threshold = 0

            if not sl.get('individual_normalize', False):
                idx_start = 0
                idx_end = len(gdata.normalized)
            s_snvalues = sorted(gdata.normalized[idx_start:idx_end])

            for val in s_snvalues:
                if val > 0:
                    s_tmp.append(val)

            # sanity check in case the normalized values were all zero
            if len(s_tmp) > 0:
                cutoff_threshold = (amax(s_snvalues)/2)-mean(s_tmp)
            else:
                cutoff_threshold = 0

            sl['cutoff_threshold'] = cutoff_threshold
            debugln("Cutoff-Threshold[%s/%s]: %s" % (gdata.name, sl['name'], str(cutoff_threshold)))

            # now filter again
            for val in gdata.normalized[prev_index:idx]:
                if (val >= cutoff_threshold):
                    gdata.events.append(val)
                else:
                    gdata.events.append(0)

            # and filter the curve (for percentages) as well
            for val in s_curve[prev_index:idx]:
                if (val >= cutoff_threshold):
                    gdata.filtered_curve.append(val)
                else:
                    gdata.filtered_curve.append(0)
            prev_index = idx

        # hack to ensure that single-slice data is in sync
        gdata.update_slice_index()

        prev_index = 0
        # calculate activity values for each slice
        for sl in gdata.slicedata:
            idx = sl['index']
            if idx == 0:
                continue
            # now count the number of events to get activity in Hz
            data = gdata.events[prev_index:idx]
            sl['frequency'] = self._count_events(data, gdata.timeline[:len(data)])
            # and get activity percantage values
            data = gdata.filtered_curve[prev_index:idx]
            sl['percentage'] = self._calculate_activity_percentage(data, gdata.timeline[:len(data)])

        # we now start to create a figure
        axfontsize = 11
        fig = figure()
        fig.suptitle(gdata.name, fontsize=20)

        ax = fig.add_subplot(4, 1, 1)
        if not original_raw_data:
            self._plot_data(ax, gdata, gdata.raw_data)
            ax.plot(gdata.timeline, gdata.regression_line, 'r.-', linewidth=0.4, markersize=0.4)
            ax.set_ylabel('Measured Signal', fontsize=axfontsize)
            ax.set_xlabel('time [s]', fontsize=axfontsize)
        else:
            self._plot_data(ax, gdata, gdata.raw_data)
            ax.plot(gdata.timeline, gdata.regression_line, 'r.-', linewidth=0.4, markersize=0.4)
            ax.set_ylabel('Measured Signal', fontsize=axfontsize)
            ax.set_xlabel('time [s]', fontsize=axfontsize)

        ax = fig.add_subplot(4,1,2)
        self._plot_data(ax, gdata, gdata.normalized)
        ax.set_ylabel('Normalized', fontsize=axfontsize)
        ax.set_xlabel('time [s]', fontsize=axfontsize)

        ax = fig.add_subplot(4,1,3)
        self._plot_data(ax, gdata, gdata.events, annotate=True, print_frequency=True)
        ax.set_ylabel('Events', fontsize=axfontsize)
        ax.set_xlabel('time [s]', fontsize=axfontsize)

        ax = fig.add_subplot(4,1,4)
        self._plot_data(ax, gdata, gdata.filtered_curve, annotate=True)
        ax.set_ylabel('Filtered Curve', fontsize=axfontsize)
        ax.set_xlabel('time [s]', fontsize=axfontsize)

        fig.savefig(os.path.join(gdata.data_dir, gdata.name + ".svg"))
        #show(fig)
        close(fig)

        # we might want to store all processed data objects for possible group-analysis at a later stage
        self.processed_data.append(gdata)

        return gdata

    def write_graph_overview(self, path):
        fig = figure()
        fig.suptitle("Raw data overview", fontsize=20)

        i = 1
        for gdata in self.processed_data:
            ax = fig.add_subplot(len(self.processed_data), 1, i)
            self._plot_data(ax, gdata, gdata.raw_data)
            ax.set_ylabel(gdata.name, fontsize=6)
            i += 1

        if path:
            fig.savefig(path)
        show(fig)
        close(fig)


    def process_dir(self, dirname):
        results = list()
        for root, dirs, files in os.walk(dirname):
            for f in files:
                if f.endswith(".csv"):
                    fname = os.path.join(root, f)
                    res = self.process_csv(fname)
                    if res != None:
                        results.append(res)

        # sort the results for nicer future analysis
        results = sorted(results, key=lambda k: k.name)
        # store detected GCaMP activity
        fname = "%s/activity-result.csv" % (dirname)
        csvfile = open(fname, 'wt')
        dwriter = csv.writer(csvfile, delimiter='\t',
            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        dwriter.writerow(['Identifier', 'Peak Frequency [Hz]', "Activity %"])
        prev_data_had_appls = False
        for gdata in results:
            if gdata.has_application_data():
                if not prev_data_had_appls:
                    dwriter.writerow(["", "", ""])
                dwriter.writerow([gdata.name, "", ""])
                for sl in gdata.slicedata:
                    dwriter.writerow([sl['name'], sl['frequency'], sl['percentage']])
                # add spacer to make it easier for humans to read
                dwriter.writerow(["", "", ""])
                prev_data_had_appls = True
            else:
                dwriter.writerow([gdata.name, gdata.slicedata[0]['frequency'], gdata.slicedata[0]['percentage']])
                prev_data_had_appls = False

######
###
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    dproc = GCaMPDataProcessor()

    data_dir_root = str(QtGui.QFileDialog.getExistingDirectory(None, "Select Directory"))
    if not data_dir_root:
        print("Abort.")
        sys.exit(1)
    print("Scan: %s" % (data_dir_root))

    dproc.process_dir(data_dir_root)
    dproc.write_graph_overview(None)

    # TODO: Add cli options for automatization
