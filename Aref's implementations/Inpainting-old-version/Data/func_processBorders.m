function diff = func_processBorders(diff)
% aaa=sum(diff.thr,2);
% bbb=sum(diff.thr,1);
% 6 rows and columns have unrelated artifacts
diff.thr(:,end-6:end) = 0;
diff.thr(1:6,:) = 0;
diff.thr(end-15:end,:) = 0;
diff.gray(1:6,:) = 0;
diff.gray(:,end-6:end) = 0;
diff.gray(end-15:end,:) = 0;
end