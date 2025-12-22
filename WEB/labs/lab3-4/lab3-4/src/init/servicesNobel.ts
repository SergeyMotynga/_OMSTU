import { ajaxNobelService } from './ajaxNobelService';
import { NobelService } from '@/services/features/nobel/service';

export const nobelService = new NobelService({ ajaxService: ajaxNobelService });