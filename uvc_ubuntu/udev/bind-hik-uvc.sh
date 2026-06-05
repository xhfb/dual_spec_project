#!/bin/sh
# 将 HIK HikCamera (2bdf:0102) 的 UVC 流接口绑定到内核 uvcvideo
set -e
BIND=/sys/bus/usb/drivers/uvcvideo/bind

[ -w "$BIND" ] || modprobe uvcvideo 2>/dev/null || true
[ -w "$BIND" ] || exit 0

for iface in /sys/bus/usb/devices/*-*:*.*/interface; do
	[ -f "$iface" ] || continue
	grep -qx "Video Streaming" "$iface" 2>/dev/null || continue
	dir="${iface%/interface}"
	vendor="$(cat "$dir/../idVendor" 2>/dev/null)" || continue
	product="$(cat "$dir/../idProduct" 2>/dev/null)" || continue
	[ "$vendor" = "2bdf" ] && [ "$product" = "0102" ] || continue
	[ -e "$dir/driver" ] && continue
	name="${dir##*/}"
	echo "$name" > "$BIND" 2>/dev/null || true
done

exit 0
