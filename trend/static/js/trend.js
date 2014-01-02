trend = function() {
        var _self = this;

        var _size = [2000, 1200], _margin = {top: 1, right: 1, bottom: 6, left: 1};

        var _canvas = d3.select("#vis")
                .append("svg")
                .attr("class","trend")
                .attr("width", _size[0])
                .attr("height", _size[1]);

        var _sankey = d3.sankey()
                .nodeWidth(0)
                .nodePadding(15)
                .size([_size[0], 300]);

        var _path = _sankey.link();

        var _area = d3.svg.area()
                .x(function(d) { return d.x; })
                .y0(function(d) { return d.y0; })
                .y1(function(d) { return d.y1; });

        var _y = d3.scale.linear().range([_size[1], 0]);

        var _x = null;

        // input data
        var _items = null, _terms = null;

        var color = d3.scale.category20();
        var formatNumber = d3.format(",.0f");
        var format = function(d) {
                return formatNumber(d) + " TWh";
        };


        this.data = function(data) {
                if(arguments.length === 0) {
                        return _items;
                }
                _items = data;
                _terms = {};
                max_sum = 0;

                _items.terms.sort(function(a, b) {
                        return b.start.year - a.start.year;
                });

                _items.terms.forEach(function(t) {
                        t.sum = 0;
                        t.year.forEach(function(tt) {
                                if ((tt.y > 2010) && (tt.d > 0)) {
                                        t.sum += tt.d;
                                }
                        });
                        if (t.sum > max_sum) {
                                max_sum = t.sum;
                        };
                        _terms[t.t] = t;
                });

                _x = buildXAxis(_items.time_slides);

                return _self;
        }

        this.layout = function() {
                _sankey.nodes(_items.nodes)
                        .links(_items.links)
                        .items(_items.terms)
                        .nodeOffset(_size[0] / _items.time_slides.length)
                        .layout(32);

                return _self;
        }

        this.render = function() {
                renderAxis(_items.time_slides);
                renderLinks();
                renderNodes();

                return _self;
        }

        this.update = function() {
                return _self;
        }

        function buildXAxis(time_slides){
                var time_slides_dict = {};
                var time_slides_offset = {};
                time_slides.forEach(function(time, i) {
                        time.sort();
                        time.forEach(function(year, j) {
                                time_slides_dict[year] = i;
                                time_slides_offset[year] = j;
                        })
                })

                var time_window = time_slides[0].length;

                var x = function(year) {
                        return (time_slides_dict[year] + ((1 / time_window) * time_slides_offset[year])) * width / time_slides.length;
                }  

                return x;
        }

        function renderAxis(time_slides){
                var axis = _canvas.selectAll(".axis")
                        .data(time_slides)
                        .enter().append("g")
                        .attr("class", "axis")
                        .attr("transform", function(d, i) {
                                return "translate(" + (i) * _size[0] / time_slides.length + "," + 0 + ")";
                        });

                axis.append("line")
                        .attr("x1", function(d) { return 0; })
                        .attr("x2", function(d) { return 0; })
                        .attr("y1", function(d) { return 0; })
                        .attr("y2", function(d) { return 1000; })
                        .style("stroke", function(d) { return "lightgray"; })
                        .style("stroke-width", function(d) { return 1; })

                axis.append("text")
                        .attr("x", -6)
                        .attr("y", 10)
                        .attr("dy", ".0em")
                        .attr("text-anchor", "end")
                        .attr("transform", null)
                        .text(function(d, i) { return d3.min(d); })
                        .attr("x", 6)
                        .attr("text-anchor", "start")
                        .style("font-weight", "bold");
        }

        function renderLinks () {
                var frame = _canvas.append("g");

                var link = frame.selectAll(".link").data(_items.links);
                        
                link.enter().append("path")
                        .attr("class", "link")
                        .attr("d", _path)
                        .style("stroke-width", function(d) { return 20 })
                        .style("fill-opacity", .6)
                        .style("fill", function(d) {
                                var key = "gradient-" + d.source_index + "-" + d.target_index;
                                _canvas.append("linearGradient")
                                        .attr("id", key)
                                        .attr("gradientUnits", "userSpaceOnUse")
                                        .attr("x1", d.source.x + 50).attr("y1", 0)
                                        .attr("x2", d.target.x).attr("y2", 0)
                                        .selectAll("stop")
                                        .data([{
                                                offset: "0%",
                                                color: color(d.source.cluster)
                                        }, {
                                                offset: "100%",
                                                color: color(d.target.cluster)
                                        }])
                                        .enter().append("stop")
                                        .attr("offset", function(d) {
                                                return d.offset;
                                        })
                                        .attr("stop-color", function(d) {
                                                return d.color;
                                        });
                                return d.color = "url(#" + key + ")";
                        })
                        .sort(function(a, b) {
                                return b.dy - a.dy;
                        });

                link.append("title").text(function(d) {
                                return d.source.name + " → " + d.target.name;
                        });
        }

        function renderNodes() {
                var frame = _canvas.append("g");

                var node = frame.selectAll(".node")
                        .data(_items.nodes)
                        .enter().append("g")
                        .attr("class", "node")
                        .attr("transform", function(d) {
                                return "translate(" + d.x + "," + d.y + ")";
                        })
                        .call(d3.behavior.drag()
                                .origin(function(d) {
                                        return d;
                                })
                                .on("dragstart", function() {
                                        this.parentNode.appendChild(this);
                                })
                                .on("drag", dragmove));

                node.append("rect")
                        .attr("height", function(d) { return d.dy; })
                        .attr("width", _sankey.nodeWidth())
                        .style("fill", function(d) { return d.color = color(d.cluster); })
                        .style("stroke", function(d) { return d.color; }) //d3.rgb(d.color).darker(2); })
                        .style("stroke-width", function(d) { return 0; })
                        .style("opacity", function(d) { return 0.6; })
                        .append("title")
                        .text(function(d) { return d.name + "\n" + format(d.value); });

                node.append("text")
                        .attr("x", -20)
                        .attr("y", function(d) { return d.dy / 2; })
                        .attr("text-anchor", "middle")
                        .attr("transform", null)
                        .text(function(d) { return d.name; })
                        .style("fill", function(d) { return "black" })//color(d.cluster); 
                        .style("font-weight", "bold")
                        .style("font", function(d) {
                                var w = d.w;
                                if (w > 15) {
                                        w = 15;
                                }
                                if (w < 10 && w > 0) {
                                        w = 10;
                                }
                                return (w) + "px sans-serif";
                        });
        }

        function dragmove(d) {
                d3.select(this).attr("transform", "translate(" + d.x + "," + (d.y = Math.max(0, Math.min(height - d.dy, d3.event.y))) + ")");
                sankey.relayout();
                link.attr("d", path);
        }


}

timeline = function() {

}

flow = function() {

}


var trend = new trend();

var q = "deep learning", start = 0, end = 10000;
d3.json("/static/js/trend_out.json", function(trend_data) {
        trend.data(trend_data).layout().render();
});